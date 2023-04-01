{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution
import Data.Type.Equality
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator )
import Data.Array.Accelerate.Pretty.Print ( Operator(..), Adoc )
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape hiding (fromIndex, toIndex, toIndex, fromIndex)
import Prelude hiding (exp)
import Data.Text.Prettyprint.Doc ( vsep, viaShow )
import Data.Array.Accelerate.Representation.Shape hiding (fromIndex, toIndex)
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Array.Accelerate.Pretty.Print (Adoc)
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Labels (LabelledArg(..))
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph (LabelledArgOp(..))
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Solver
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph
import Data.Array.Accelerate.Analysis.Hash.Exp (intHost, hashQ, encodeScalarType, encodeScalarConst)

import Data.Accelerate.TensorFlow.Type
import Proto.Tensorflow.Core.Util.TestLog (TestResults'BenchmarkType(TestResults'BenchmarkType'Unrecognized))
import qualified TensorFlow.Types as TF
import Data.ByteString.Builder (Builder)

data TensorOp op where
  TConstant :: TF.OneOf TFAll a => ScalarType a -> a -> TensorOp (Out sh a -> ())
  TId :: TF.OneOf TFAll a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())

  TAdd :: TF.OneOf TFNum a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TMul :: TF.OneOf TFNum a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TSub :: TF.OneOf TFNum a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TNeg :: TF.OneOf TFNeg a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TAbs :: TF.OneOf TFAbs a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())
  TSign :: TF.OneOf TFSign a => ScalarType a -> TensorOp (In sh a -> Out sh a -> ())

  TTruncateDiv :: TF.OneOf TFNum a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())
  TTruncateMod :: TF.OneOf TFTruncateMod a => ScalarType a -> TensorOp (In sh (a, a) -> Out sh a -> ())

  TWhere :: ScalarType a -> TensorOp (In sh a -> Out DIM1 sh -> ())
  TTensorScatter :: ScatterFun -> TensorOp (Mut sh' s -> In sh sh' -> In sh s -> ())
  TBooleanMask :: ScalarType s -> TensorOp (In DIM1 s -> In DIM1 PrimBool -> Out DIM1 s -> ())
  TSelect :: TensorOp (In sh Bool -> In sh t -> In sh t -> Out sh t -> ())

instance EncodeOperation TensorOp where
  encodeOperation :: TensorOp t -> Builder
  encodeOperation t = case t of
    TConstant st s -> intHost $(hashQ ("Constant" :: String)) <> encodeScalarConst st s
    TId st -> intHost $(hashQ ("Id" :: String)) <> encodeScalarType st
    TAdd st -> intHost $(hashQ ("Add" :: String)) <> encodeScalarType st
    TMul st -> intHost $(hashQ ("Mul" :: String)) <> encodeScalarType st
    TSub st -> intHost $(hashQ ("Sub" :: String)) <> encodeScalarType st
    TNeg st -> intHost $(hashQ ("Neg" :: String)) <> encodeScalarType st
    TAbs st -> intHost $(hashQ ("Abs" :: String)) <> encodeScalarType st
    TSign st -> intHost $(hashQ ("Sign" :: String)) <> encodeScalarType st
    TTruncateDiv st -> intHost $(hashQ ("TruncateDiv" :: String)) <> encodeScalarType st
    TTruncateMod st -> intHost $(hashQ ("TruncateMod" :: String)) <> encodeScalarType st
    TWhere st -> intHost $(hashQ ("Where" :: String)) <> encodeScalarType st
    TTensorScatter sf -> encodeScatterFun sf
    TBooleanMask st -> intHost $(hashQ ("BooleanMask" :: String)) <> encodeScalarType st
    TSelect -> intHost $(hashQ ("Select" :: String))

instance PrettyOp TensorOp where
  prettyOp :: TensorOp t -> Adoc
  prettyOp (TConstant s e)    = vsep ["TConst", prettyConst (TupRsingle s) e]
  prettyOp (TId s)            = vsep ["TId", prettyScalarType s]
  prettyOp (TWhere s)         = vsep ["TWhere", prettyScalarType s]
  prettyOp (TTensorScatter f) = vsep ["TTensorScatter", viaShow f]
  prettyOp (TBooleanMask s)   = vsep ["TBooleanMask", prettyScalarType s]
  prettyOp TSelect            = vsep ["TSelect"]
  prettyOp _                  = vsep ["pretty me!"]

instance NFData' TensorOp where
  rnf' !_ = ()

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkMapF (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate f (ArgArray _ (ArrayR sh t) gv gvb) = undefined
  -- mkGenerate f (ArgArray _ (ArrayR sh t) gv gvb) Vraag woensdag: hoe kan ik die type family gebruiken om Int als Int64 te beschouwen?
  --   | DeclareVars lhs w k       <- declareVars $ buffersR (TupRsingle scalarTypeInt)
  --   , DeclareVars lhs' w' k'    <- declareVars $ buffersR (shapeType sh)
  --   , DeclareVars lhs'' w'' k'' <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
  --   = -- 1) Create a Tensor of shape sh with only ones
  --     aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (fromGrounds gv)) $
  --     Alet (LeftHandSideWildcard TupRunit) TupRunit
  --     (Exec
  --       (TConstant scalarTypeInt 1)
  --       (ArgArray Out (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars w gv) (k weakenId) :>: ArgsNil)
  --     ) $
  --     -- 2) Obtain a 1D array of indexes (tf.where returns list of indices pointing to values > 0)
  --     aletUnique lhs' (desugarAlloc (ArrayR sh (shapeType sh)) (weakenVars w $ fromGrounds gv)) $
  --     aletUnique lhs'' (Compute (ShapeSize sh (paramsIn' $ weakenVars (w' .> w) $ fromGrounds gv))) $
  --     Alet (LeftHandSideWildcard TupRunit) TupRunit
  --     (Exec
  --       (TWhere scalarTypeInt)
  --       (ArgArray In (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w'' .> w' .> w) gv) (k (w'' .> w')) :>:
  --        ArgArray Out (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k'' weakenId)) (k' w'') :>:
  --        ArgsNil)
  --     )
  --     -- 3) Apply map on array of indices.
  --     (mkMap
  --       (weaken (w'' .> w' .> w) f)
  --       (ArgArray In (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k'' weakenId)) (k' w''))
  --       (ArgArray Out (ArrayR dim1 t) (TupRpair TupRunit (k'' weakenId)) (weakenVars (w'' .> w' .> w) gvb)))

  mkPermute
    (ArgFun comb)
    (ArgArray _ (ArrayR sh' _) gv' gvb')
    perm
    (ArgArray _ (ArrayR sh t) gv gvb)
    | maybeSh'                     <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh'))
    , DeclareVars lhs w k          <- declareVars $ buffersR maybeSh'
    , DeclareVars lhs' w' k'       <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (shapeType sh')
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR t
    = -- 1) Create an array of maybeSh' with perm
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh') (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh') (weakenVars w gv) (k weakenId))
      ) $
      -- 2) To apply boolean mask with 1D arrays, we need to flatten. Calculate the dim1 size.
      aletUnique lhs' (Compute (ShapeSize sh (paramsIn' $ weakenVars w $ fromGrounds gv))) $
      -- 3) Get 1D array of (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs'' (desugarAlloc (ArrayR sh (shapeType sh')) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask (shapeType sh')
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' w'')) (isJust (TupRpair TupRunit (shapeType sh')) (k (w'' .> w'))))
        (fromJust (shapeType sh') (k (w'' .> w')))
        (k'' weakenId)
      ) $
      -- 3) Get 1D array of source indices (sh) with perm output (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs''' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w'' .> w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask t
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' (w''' .> w''))) (isJust (TupRpair TupRunit (shapeType sh')) (k (w''' .> w'' .> w'))))
        (weakenVars (w''' .> w'' .> w' .> w) gvb)
        (k''' weakenId)
      ) $
      -- 4) Apply tf.tensor_scatter
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR sh' t) (weakenVars (w''' .> w'' .> w' .> w) gv') (weakenVars (w''' .> w'' .> w' .> w) gvb') :>:
        ArgArray In (ArrayR sh (shapeType sh')) (weakenVars (w''' .> w'' .> w' .> w) gv) (k'' w''') :>:
        ArgArray In (ArrayR sh t) (weakenVars (w''' .> w'' .> w' .> w) gv) (k''' weakenId) :>:
        ArgsNil
      )
        where scatterFun = case comb of
                Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp fun (Pair (Evar (Var _ (SuccIdx ZeroIdx))) (Evar (Var _ ZeroIdx)))))) -> case fun of
                  PrimAdd _ -> ScatterFunAdd
                  PrimSub _ -> ScatterFunMin
                  _         -> error "primfun not yet supported"
                _ -> error "complex combination for permute not supported"

instance SimplifyOperation TensorOp where

instance SLVOperation TensorOp where
  slvOperation _ = Nothing

instance ShrinkArg (BackendClusterArg TensorOp) where
  shrinkArg s NoFusionArg = NoFusionArg
  deadArg NoFusionArg = NoFusionArg

instance NFData' (BackendClusterArg TensorOp) where
  rnf' NoFusionArg = ()

instance MakesILP TensorOp where
  type BackendVar TensorOp = ()
  type BackendArg TensorOp = ()
  data BackendClusterArg TensorOp arg = NoFusionArg
  mkGraph _ _ _ = mempty
  labelLabelledArg _ _ (L arg l) = LOp arg l ()
  getClusterArg (LOp _ _ ()) = NoFusionArg
  finalize = mempty
  encodeBackendClusterArg NoFusionArg = intHost $(hashQ ("NoFusionArg" :: String))

booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) aOut@(ArgArray _ _ gv _) gvbIn gvbOut = Exec (TBooleanMask s) (ArgArray In (ArrayR dim1 t) gv gvbIn :>: aOut :>: ArgArray Out (ArrayR dim1 t) gv gvbOut :>: ArgsNil)
booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
isJust _ (TupRpair word8 _) = word8
isJust _ _ = error "impossible"

fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
fromJust _ (TupRpair _ (TupRpair _ a)) = a
fromJust _ _ = error "impossible"

mkMapF :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t
  -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut | TensorDict <- tfAllDict s = Exec (TConstant s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut = mkMapPrimAppF f env exp aOut

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkMapF
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkMapF env (Evar (Var s idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer s
  , TensorDict <- tfAllDict s
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          (TId s)
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkMapF env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkMapF env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

mkMapF env (ToIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ shapeType sh
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (toIndex sh (k w') (k' weakenId)))) aOut

mkMapF env (ShapeSize sh exp) aOut
  | DeclareVars lhs _ k <- declareVars $ shapeType sh
  = mkMapF env (Let lhs exp (shapeSize sh (k weakenId))) aOut

mkMapF env (FromIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ TupRsingle scalarTypeInt
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (fromIndex sh (k w') (k' weakenId)))) aOut

-- TODO
mkMapF _ (ArrayInstr arr exp) _ = case arr of
  Index (Var groundR idx) -> undefined
  Parameter (Var s idx)   -> undefined

mkMapF _ (Foreign _ _ _ _) _ = undefined
mkMapF _ Nil _ = Return TupRunit
mkMapF _ (VecPack _ _) _ = undefined
mkMapF _ (VecUnpack _ _) _ = undefined
mkMapF _ (IndexSlice _ _ _) _ = undefined
mkMapF _ (IndexFull _ _ _) _ = undefined

mkMapF _ (Case _ _ _) _ = undefined

mkMapF env (Cond cond exp1 exp2) (ArgArray _ (ArrayR sh t) gv gvb) = undefined

mkMapF _ (While _ _ _) _ = undefined
mkMapF _ (PrimConst _) _ = undefined

mkMapF _ (Undef _) _ = undefined
mkMapF _ (Coerce _ _ _) _ = undefined
mkMapF _ _ _ = error "impossible"

mkMapPrimAppF :: PrimFun (a -> t) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapPrimAppF (PrimAdd nt)  | TensorDict <- tfNumDict nt = mkMapPrimAppF' $ TAdd (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimMul nt)  | TensorDict <- tfNumDict nt = mkMapPrimAppF' $ TMul (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimSub nt)  | TensorDict <- tfNumDict nt = mkMapPrimAppF' $ TSub (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimNeg nt)  | TensorDict <- tfNegDict nt = mkMapPrimAppF' $ TNeg (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimAbs nt)  | TensorDict <- tfNegDict nt = mkMapPrimAppF' $ TAbs (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimSig nt)  | TensorDict <- tfNegDict nt = mkMapPrimAppF' $ TSign (SingleScalarType (NumSingleType nt))

mkMapPrimAppF (PrimQuot it) | TensorDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TTruncateDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimRem it)  | TensorDict <- tfTruncateModDict (IntegralNumType it) = mkMapPrimAppF' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))

mkMapPrimAppF _ = undefined

mkMapPrimAppF' :: TensorOp (In sh a -> Out sh b -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh b) -> OperationAcc TensorOp env ()
mkMapPrimAppF' op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

shapeSize :: ShapeR sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
shapeSize ShapeRz TupRunit                                = Const scalarTypeInt 1
shapeSize (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz))
  = PrimApp (PrimMul (IntegralNumType TypeInt))
      (Pair
        (shapeSize shr sh)
        (Evar sz))
shapeSize _ _                                             = error "impossible"

toIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
toIndex ShapeRz TupRunit TupRunit = Const scalarTypeInt 0
toIndex (ShapeRsnoc shr) (TupRpair sh _) (TupRpair ix (TupRsingle i))
  = PrimApp (PrimAdd (IntegralNumType TypeInt))
      (Pair
        (toIndex shr sh ix)
        (PrimApp (PrimMul (IntegralNumType TypeInt))
          (Pair
            (shapeSize shr sh)
            (Evar i))))
toIndex _ _ _                     = error "impossible"

fromIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' Int -> PreOpenExp (ArrayInstr env) env' sh
fromIndex ShapeRz TupRunit _                                = Nil
fromIndex (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz)) (TupRsingle i)
  | DeclareVars lhs w k <- declareVars $ TupRsingle scalarTypeInt
  = Pair
    (Let 
      lhs 
      (PrimApp (PrimQuot TypeInt) (Pair (Evar i) (Evar sz))) 
      (fromIndex shr (weakenVars w sh) (k weakenId)))
    (case shr of
      ShapeRz -> Evar i
      _       -> PrimApp (PrimRem TypeInt) (Pair (Evar i) (Evar sz))
    )

fromIndex _ _ _              = error "impossible"

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a))

instance Sink BufferIdx where
  weaken w (BIdx idx) = BIdx (weaken w idx)

type BufferEnv benv env = Env (BufferIdx benv) env

weakenEnv :: Sink f => (env1 :> env') -> Env (f env1) env2 -> Env (f env') env2
weakenEnv w = mapEnv (weaken w)

distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"

data ScatterFun where
  ScatterFunAdd :: ScatterFun
  ScatterFunMin :: ScatterFun
  deriving Show

encodeScatterFun :: ScatterFun -> Builder
encodeScatterFun ScatterFunAdd = intHost $(hashQ ("ScatterFunAdd" :: String))
encodeScatterFun ScatterFunMin = intHost $(hashQ ("ScatterFunMin" :: String))

prettyScatterFun :: ScatterFun -> Adoc
prettyScatterFun ScatterFunAdd = "ScatterFunAdd"
prettyScatterFun ScatterFunMin = "ScatterFunMin"

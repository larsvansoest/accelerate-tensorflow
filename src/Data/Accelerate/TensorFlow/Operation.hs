{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Exp
import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Analysis.Hash.Exp
import Data.Array.Accelerate.Analysis.Hash.Operation
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph hiding (Var)
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Labels
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.Representation.Shape (shapeType, ShapeR)
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution
import Data.Type.Equality

data TensorOp op where
  TConst :: ScalarType s -> s -> TensorOp (Out sh s -> ())
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())

instance PrettyOp TensorOp where
  prettyOp (TConst _ _) = "TTensor"
  prettyOp (TPrimFun _) = "TBinOp"

instance NFData' TensorOp where
  rnf' !_ = ()

mapXTimesTwoPlusOne' :: forall env sh s t. Arg env (In sh Int64) -> Arg env (Out sh Int64) -> OperationAcc TensorOp env ()
mapXTimesTwoPlusOne' (ArgArray _ arrayR@(ArrayR sh ty) gvIn gvbIn) argOut
  | DeclareVars lhs  w  k  <- declareVars $ buffersR ty
  , DeclareVars lhs' w' k' <- declareVars $ buffersR ty
  = let
    nInt64 :: NumType Int64
    nInt64 = IntegralNumType TypeInt64
    sInt64 :: ScalarType Int64
    sInt64 = SingleScalarType (NumSingleType nInt64)
    bInt64 :: GroundR (Buffer Int64)
    bInt64 = GroundRbuffer sInt64
    arrayR' :: ArrayR (Array sh (Int64, Int64))
    arrayR' = ArrayR sh $ TupRpair (TupRsingle sInt64) (TupRsingle sInt64)
  in aletUnique lhs (desugarAlloc arrayR (fromGrounds gvIn)) $
     Alet (LeftHandSideWildcard TupRunit) TupRunit
       (Exec (TConst sInt64 2) (ArgArray Out (ArrayR sh (TupRsingle sInt64)) (weakenVars w gvIn) (weakenVars w gvbIn) :>: ArgsNil)) $
         Alet (LeftHandSideWildcard TupRunit) TupRunit
           (Exec (TPrimFun (PrimMul nInt64)) (ArgArray In arrayR' (weakenVars w gvIn) (TupRpair (weakenVars w gvbIn) (k weakenId)) :>: weaken w argOut :>: ArgsNil)) $
             aletUnique lhs' (desugarAlloc arrayR (fromGrounds (weakenVars w gvIn))) $
               Alet (LeftHandSideWildcard TupRunit) TupRunit
                 (Exec (TConst sInt64 1) (ArgArray Out (ArrayR sh (TupRsingle sInt64)) (weakenVars (w' .> w) gvIn) (weakenVars (w' .> w) gvbIn) :>: ArgsNil)) $
                   Exec (TPrimFun (PrimAdd nInt64)) (ArgArray In arrayR' (weakenVars (w' .> w) gvIn) (TupRpair (weakenVars (w' .> w) gvbIn) (k' weakenId)) :>: weaken (w' .> w) argOut :>: ArgsNil)

mapXTimesTwoPlusOne :: forall env sh. Arg env (In sh Int64) -> Arg env (Out sh Int64) -> OperationAcc TensorOp env ()
mapXTimesTwoPlusOne (ArgArray _ arrayR@(ArrayR sh _) gvIn gvbIn) argOut = let
  nInt64 :: NumType Int64
  nInt64 = IntegralNumType TypeInt64

  sInt64 :: ScalarType Int64
  sInt64 = SingleScalarType (NumSingleType nInt64)

  bInt64 :: GroundR (Buffer Int64)
  bInt64 = GroundRbuffer sInt64

  arrayR' :: ArrayR (Array sh (Int64, Int64))
  arrayR' = ArrayR sh $ TupRpair (TupRsingle sInt64) (TupRsingle sInt64)

  gvIn' :: TupR (Var GroundR (env, Buffer Int64)) sh
  gvIn' = mapTupR (weaken (weakenSucc weakenId)) gvIn

  gvIn'' :: TupR (Var GroundR ((env, Buffer Int64), Buffer Int64)) sh
  gvIn'' = mapTupR (weaken (weakenSucc weakenId)) gvIn'

  varToI0 :: forall env. TupR (Var GroundR (env, Buffer Int64)) (Buffer Int64)
  varToI0 = TupRsingle $ Var bInt64 ZeroIdx

  gvbIn' :: TupR (Var GroundR (env, Buffer Int64)) (Buffer Int64, Buffer Int64)
  gvbIn' = TupRpair (mapTupR (weaken (weakenSucc weakenId)) gvbIn) varToI0

  gvbIn'' :: TupR (Var GroundR ((env, Buffer Int64), Buffer Int64)) (Buffer Int64, Buffer Int64)
  gvbIn'' = TupRpair (mapTupR (weaken (weakenSucc weakenId)) (mapTupR (weaken (weakenSucc weakenId)) gvbIn)) varToI0

  argOut' :: Arg (env, Buffer Int64) (Out sh Int64)
  argOut' = weaken (weakenSucc weakenId) argOut
  in -- eerst nieuwe buffer aanmaken, eerst array aanmaken van zelfde grootte
    Alet -- kan je gebruiken voor nieuwe variabelen of side effects uitvoeren en dan doorgaan met iets anders
    (LeftHandSideSingle bInt64) -- variable introduceren
    (TupRsingle Unique) -- uniqueness van nieuwe variabele
    (Alloc sh sInt64 (groundToExpVar (shapeType sh) gvIn))
    -- array vullen met tweeën
    $ Alet (LeftHandSideWildcard TupRunit)
      TupRunit
      (Exec (TConst sInt64 2) (ArgArray Out arrayR gvIn' varToI0 :>: ArgsNil)) -- (TupRsingle $ Var int64Buffer ZeroIdx) refereert naar een array
      -- keer twee
      $ Alet (LeftHandSideWildcard TupRunit)
        TupRunit
        (Exec
          (TPrimFun (PrimMul nInt64))
          (ArgArray In arrayR' gvIn' gvbIn' :>: argOut' :>: ArgsNil)
        )
        -- nieuwe array aanmaken van zelfde grootte
        $ Alet
            (LeftHandSideSingle bInt64)
            (TupRsingle Unique)
            (Alloc sh sInt64 (groundToExpVar (shapeType sh) gvIn'))
            -- array vullen met 1'en
            $ Alet (LeftHandSideWildcard TupRunit)
              TupRunit
              ( Exec
                (TConst sInt64 1)
                (ArgArray Out arrayR gvIn'' varToI0 :>: ArgsNil)
              )
              -- plus 1
              $ Exec
                  (TPrimFun (PrimAdd nInt64))
                  (ArgArray In arrayR' gvIn'' gvbIn''
                    :>: weaken (weakenSucc weakenId) argOut'
                    :>: ArgsNil
                  )

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) _ aOut = mkMapF _ body aOut 
  -- lhs nodig bij het gebruik van een variabele (e.g. Let, Evar), lhs geef je Env mee
  mkMap _ _ _ = error "impossible"

  mkGenerate = undefined
  mkPermute = undefined

data BufferIdx benv a where
  BufferIdx :: Idx benv (Buffer a) -> BufferIdx benv a



-- lhsToEnv (LeftHandSideSingle s) (TupRsingle (Var _ idx))
--   | Refl <- reprIsSingle @ScalarType @a @Buffer s
--   = Push Empty (BufferIdx idx)

-- lhsToEnv (LeftHandSidePair l1 l2) (TupRpair t1 t2)
--   = lhsToEnv l1 t1 `concatBufferIdx` lhsToEnv l2 t2

-- lhsToEnv (LeftHandSideWildcard _) _ = undefined

-- lhsToEnv _ _ = error "impossible"




concatBufferIdx :: Env (BufferIdx benv) a -> Env (BufferIdx benv) b -> Env (BufferIdx benv) b
concatBufferIdx = undefined

mkMapF :: forall env env' benv sh t. Env (BufferIdx benv) env' -> PreOpenExp (ArrayInstr env) env' t -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut = Exec (TConst s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit (mkMapF env (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))) $
   Exec (TPrimFun f) (ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>: weaken w aOut :>: ArgsNil)

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh t) gv gvb)
 | a1 <- expType exp1
 , a2 <- expType exp2
 , DeclareVars lhs1 w1 k1 <- declareVars $ buffersR a1
 , DeclareVars lhs2 w2 k2 <- declareVars $ buffersR a2
 = aletUnique lhs1 (desugarAlloc (ArrayR sh a1) (fromGrounds gv)) $ 
   Alet (LeftHandSideWildcard TupRunit) TupRunit (mkMapF env (weakenArrayInstr w1 exp1) (ArgArray Out (ArrayR sh a1) (weakenVars w1 gv) (k1 weakenId))) $ 
   aletUnique lhs2 (desugarAlloc (ArrayR sh a2) (fromGrounds (weakenVars w1 gv))) $ 
   mkMapF (lhsToEnv env elhs _) (weakenArrayInstr (w2 .> w1) exp2) (ArgArray Out (ArrayR sh a2) (weakenVars (w2 .> w1) gv) (k2 weakenId))

mkMapF _ _ _ = undefined

lhsToEnv :: forall a env env' benv sh.
  Env (BufferIdx benv) env
  -> ELeftHandSide a env env'
  -> TupR (Var GroundR benv) (Buffers a) 
  -> Env (BufferIdx benv) env'
lhsToEnv env (LeftHandSideSingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @a @Buffer s
  = Push env (BufferIdx idx)
lhsToEnv env (LeftHandSidePair l1 l2) (TupRpair t1 t2) = lhsToEnv (lhsToEnv env l1 t1) l2 t2
lhsToEnv env (LeftHandSideWildcard _) _ = env
lhsToEnv _ _ _ = error "impossible"

-- let evar, pair nil, play around with zipwith or fold

-- 0.1.2.3.4.5.6

-- mkMapBody (PrimApp f exp) aOut
--  | DeclareVars lhs w k  <- declareVars $ buffersR ty 
--  = aletUnique lhs (mkMapBody exp aOut) $ -- alloc gebruiken -> output voor mkmap body exp -> invoer voor volgende exec
--    Exec (TPrimFun f) $ ArgArray In _ _ _ :>: aOut :>: ArgsNil
-- mkMapBody _ _ _ = undefined



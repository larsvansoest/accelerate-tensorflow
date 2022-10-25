{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ViewPatterns         #-}

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
import Data.Array.Accelerate.Representation.Shape (shapeType)
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution

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

-- voor het dynamisch toevoegen van nieuwe variabelen: zie Trafo/Vars.hs => declareVars, gebruik pattern matching in guards
-- e.g.  | DeclareVars lhs w k <- declareVars $ buffersR (TupRpair ty1 ty2) =

-- mkMapF :: Arg env (Fun' (s -> t)) -> Arg env (In sh s) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
-- Constant values
-- mkMapF (ArgFun (Body (Const s e)))   _ _ = Exec (TConst s e) ArgsNil
-- mkMapF (ArgFun (Body (PrimConst _))) _ _ = undefined
-- -- Primitive scalar operations
-- mkMapF (ArgFun (Body (PrimApp f exp))) (ArgArray m (ArrayR sh tR) g b) o
--   = Exec (TPrimFun f) (ArgArray m (ArrayR sh _) g _ :>: _ :>: ArgsNil)

-- mkMapF (ArgFun (Body (ArrayInstr _ _))) input output = undefined

-- Others
-- mkMapF (ArgFun (Body (Let _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Evar _))) input output = undefined
-- mkMapF (ArgFun (Body (Foreign _ _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (VecUnpack _ _))) input output = undefined
-- mkMapF (ArgFun (Body (IndexSlice _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (IndexFull _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (FromIndex _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Case _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Cond _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (While _ _ _))) input output = undefined
-- mkMapF (ArgFun (Body (Undef _))) input output = undefined
-- mkMapF (ArgFun (Body (Coerce _ _ _))) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideSingle (SingleScalarType (NumSingleType (IntegralNumType _)))) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideSingle (SingleScalarType (NumSingleType (FloatingNumType _)))) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideSingle (VectorScalarType _)) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSideWildcard _) f)) input output = undefined
-- mkMapF (ArgFun (Lam (LeftHandSidePair _ _) f)) input output = undefined

-- -- Redundant pattern matches
-- mkMapF (ArgFun (Body (Pair _ _))) input output = undefined
-- mkMapF (ArgFun (Body Nil)) input output = undefined
-- mkMapF (ArgFun (Body (VecPack _ _))) input output = undefined
-- mkMapF (ArgFun (Body (ShapeSize _ _))) input output = undefined
-- mkMapF (ArgFun (Body (ToIndex _ _ _))) input output = undefined

instance DesugarAcc TensorOp where
  mkMap = undefined
  mkGenerate = undefined
  mkPermute = undefined
